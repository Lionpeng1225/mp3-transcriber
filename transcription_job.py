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
from adaptive_decoding import AdaptiveDecodingPipeline
from opencc import OpenCC
from audio_chunks import SAMPLE_RATE, audio_chunks
from parallel_segments import central_silence, transcribe_pair
from audio_cache import AudioCache
from job_storage import save_job, read_job
from resource_scheduler import inference_slot
from segment_boundaries import owned_segment

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


class SpeakerPaused(Exception):
    pass


class TranscriptionJob:
    def __init__(self, recording, size, language, mode, terms, backend=None, parallel_enabled=False,
                 speaker_timing="manual", speaker_token="", word_timestamps=True):
        self.id = uuid4().hex
        self.name = Path(recording.name).stem
        self.size, self.language, self.mode, self.terms = size, language, mode, terms
        self.backend = backend or {"engine": "faster-whisper"}
        self.parallel_enabled = bool(parallel_enabled)
        self.chunk_parallel_enabled = None
        self.word_timestamps = bool(word_timestamps)
        self.chunk_word_timestamps = None
        self.speaker_timing = speaker_timing
        self._speaker_token = speaker_token
        self.condition = Condition()
        self.pause_requested = False
        self.stop_requested = False
        self.status = "running"
        self.phase = "等待載入模型"
        self.rows = []
        self.words = []
        self.speaker_parts = []
        self.speaker_profiles = {}
        self.content_revision = 0
        self._export_cache = None
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
        self.speaker_labels = []
        self.speaker_done = 0
        self.speaker_revision = 0
        self.speaker_target = 0
        self.speaker_run_mode = speaker_timing
        self.speaker_status = "not_started"
        self.speaker_error = ""
        self.directory = STORE_ROOT / self.id
        self.directory.mkdir(parents=True)
        extension = Path(recording.name).suffix.lower()
        if extension not in {".mp3", ".mp4"}:
            extension = ".mp3"
        self.source = self.directory / ("recording" + extension)
        self.source.write_bytes(recording.getbuffer())
        self.audio_cache = AudioCache(self.source)
        self.save()

    def save(self, force=True):
        with self.condition:
            now = perf_counter()
            if (not force and self.last_saved_at is not None
                    and now - self.last_saved_at < PROGRESS_SAVE_INTERVAL):
                return False
            data = self.snapshot(include_rows=False)
            data.update(id=self.id, source_filename=self.source.name,
                        size=self.size, language=self.language, backend=self.backend,
                        mode=self.mode, terms=self.terms, word_timestamps=self.word_timestamps,
                        resume_offset=self.resume_offset, committed_rows=self.committed_rows)
            data.update(speaker_status=self.speaker_status, speaker_profiles=self.speaker_profiles,
                        speaker_error=self.speaker_error, speaker_timing=self.speaker_timing)
            if not (self.directory / "progress.sqlite3").exists() and (self.directory / "state.json").exists():
                import shutil
                shutil.copy2(self.directory / "state.json", self.directory / "state.legacy.json")
            save_job(self, data)
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
        data = read_job(directory)
        job = cls.__new__(cls)
        for key in ("name", "size", "language", "mode", "terms", "rows", "processed",
                    "resume_offset", "committed_rows", "error"):
            setattr(job, key, data[key])
        job.id = job_id
        job.backend = data.get("backend", {"engine": "faster-whisper"})
        job.parallel_enabled = data.get("parallel_enabled", False)
        job.chunk_parallel_enabled = None
        job.word_timestamps = data.get("word_timestamps", True)
        job.chunk_word_timestamps = None
        job.directory = directory
        source_filename = data.get("source_filename", "recording.mp3")
        if source_filename not in {"recording.mp3", "recording.mp4"}:
            raise ValueError("保存工作中的媒體檔名無效")
        job.source = directory / source_filename
        job.audio_cache = AudioCache(job.source)
        job.words = data.get("words", [[] for _ in job.rows])
        job.speaker_parts = data.get("speaker_parts", [])
        job.speaker_profiles = data.get("speaker_profiles", {})
        job.content_revision = 0
        job._export_cache = None
        job.condition = Condition()
        job.pause_requested = job.stop_requested = False
        job.status = data["status"] if data["status"] in TERMINAL else "interrupted"
        job.phase = "已載入自動保存的內容"
        job.last_saved_at = perf_counter()
        job.speaker_labels = data.get("speaker_labels")
        job.speaker_status = data.get("speaker_status", "not_started")
        job.speaker_error = data.get("speaker_error", "")
        job.speaker_timing = data.get("speaker_timing", "manual")
        job.speaker_done = min(data.get("speaker_done", len(job.speaker_labels or [])), len(job.rows))
        job.speaker_labels = list(job.speaker_labels or [])[:job.speaker_done]
        job.speaker_parts = (job.speaker_parts + [[] for _ in range(job.speaker_done)])[:job.speaker_done]
        job._stored_rows = 0
        job._stored_labels = 0
        job.speaker_revision = 0
        job.speaker_target = job.speaker_done
        job.speaker_run_mode = job.speaker_timing
        job._speaker_token = ""
        if job.speaker_status in {"running", "paused", "waiting"}:
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
            self.elapsed_before = self.snapshot(include_rows=False)["elapsed"]
            self.elapsed_paused_at = None
            self.paused_seconds = 0
            self.speaker_revision += 1
            self.speaker_done = min(self.speaker_done, self.committed_rows)
            self.speaker_labels = list(self.speaker_labels or [])[:self.speaker_done]
            self.speaker_target = self.speaker_done
            self.rows = self.rows[:self.committed_rows]
            self.words = self.words[:self.committed_rows]
            self.speaker_parts = self.speaker_parts[:self.speaker_done]
            self._stored_rows = min(getattr(self, "_stored_rows", 0), self.committed_rows)
            self._stored_labels = min(getattr(self, "_stored_labels", 0), self.speaker_done)
            self.content_revision += 1
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
            self.save()
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
                    if self.speaker_timing == "pause":
                        self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())
                else:
                    self.status = "pausing"
                self.condition.notify_all()

    def set_parallel_enabled(self, enabled):
        with self.condition:
            if getattr(self, "deleted", False):
                raise ValueError("此工作已刪除")
            self.parallel_enabled = bool(enabled)
            self.save()

    def set_word_timestamps(self, enabled):
        with self.condition:
            if getattr(self, "deleted", False):
                raise ValueError("此工作已刪除")
            self.word_timestamps = bool(enabled)
            self.save()

    def set_speaker_timing(self, timing, token=""):
        if timing not in {"manual", "during", "pause", "after"}:
            raise ValueError("無效的說話者辨識時機")
        with self.condition:
            self.speaker_timing = timing
            self.speaker_run_mode = timing
            self.speaker_target = len(self.rows)
            if token:
                self._speaker_token = token
            self.condition.notify_all()
            self.save()
            if (timing == "during" or (timing == "pause" and self.status == "paused")
                    or (timing == "after" and self.status == "completed")):
                self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())

    def start_speaker_diarization(self, token):
        with self.condition:
            if len(self.rows) <= self.speaker_done:
                raise ValueError("目前沒有尚未辨識說話者的已轉錄片段")
            self._launch_speaker_diarization(token, mode="manual")

    def _load_saved_speaker_token(self):
        try:
            from speaker_token_store import load_token
            return load_token(Path(__file__).resolve().parent / ".speaker-token.dpapi")
        except OSError:
            return ""

    def _launch_speaker_diarization(self, token, mode=None):
        with self.condition:
            self.speaker_run_mode = mode or self.speaker_timing
            self.speaker_target = len(self.rows)
            self.condition.notify_all()
            if getattr(self, "speaker_thread", None) and self.speaker_thread.is_alive():
                return
            if self.speaker_run_mode != "during" and self.speaker_target <= self.speaker_done:
                return
            if not token:
                self.speaker_status = "failed"
                self.speaker_error = "請先輸入或保存 Hugging Face Read 權杖。"
                self.save()
                return
            self.speaker_status = "running"
            self.speaker_error = ""
            self.save()
            self.speaker_thread = Thread(target=self._run_speaker_diarization,
                                         args=(token,), daemon=True)
            self.speaker_thread.start()

    def _speaker_checkpoint(self, nonblocking=False):
        with self.condition:
            while self.speaker_run_mode == "during" and (
                    self.pause_requested or self.status in {"paused", "pausing"}) and self.status not in TERMINAL:
                self.speaker_status = "paused"
                self.speaker_phase = "說話者辨識已暫停，等待繼續轉錄"
                if nonblocking:
                    raise SpeakerPaused()
                self.condition.wait(timeout=1)
            if self.speaker_run_mode == "during" and (self.stop_requested or self.status in TERMINAL - {"completed"}):
                raise Stopped()
            self.speaker_status = "running"

    def _run_speaker_diarization(self, token):
        try:
            from speaker_diarization import diarize_batch
            self.audio_cache.ensure(self._speaker_checkpoint)
            while True:
                self._speaker_checkpoint()
                with self.condition:
                    target = len(self.rows) if self.speaker_run_mode == "during" else self.speaker_target
                    if self.speaker_done >= target:
                        if self.speaker_run_mode == "during" and self.status not in TERMINAL:
                            self.speaker_status = "waiting"
                            self.speaker_phase = "等待新的已轉錄片段"
                            self.condition.wait(timeout=1)
                            continue
                        self.speaker_status = "completed"
                        self.speaker_phase = "本次已轉錄範圍的說話者辨識完成"
                        self.save()
                        self.speaker_thread = None
                        return
                    begin = self.speaker_done
                    # In synchronous mode, collect a completed audio chunk before
                    # clustering instead of assigning identities to isolated words.
                    if (self.speaker_run_mode == "during" and self.status not in TERMINAL
                            and self.resume_offset <= self.rows[begin][0]
                            and self.rows[target - 1][1] - self.rows[begin][0] < 30):
                        self.speaker_status = "waiting"
                        self.speaker_phase = "等待目前轉錄分段完成後分析說話者"
                        self.condition.wait(timeout=1)
                        continue
                    limit = self.rows[begin][0] + 30
                    finish = begin + 1
                    while finish < target and self.rows[finish][1] <= limit:
                        finish += 1
                    batch = list(self.rows[begin:finish])
                    batch_words = list(self.words[begin:finish])
                    profiles = dict(self.speaker_profiles)
                    context_end = self.rows[target - 1][1]
                    revision = self.speaker_revision
                    self.speaker_phase = f"分析 {timestamp(batch[0][0])}–{timestamp(batch[-1][1])}"
                def progress(message):
                    with self.condition:
                        self.speaker_phase = f"{timestamp(batch[0][0])}–{timestamp(batch[-1][1])}：{message}"
                try:
                    labels, parts, profiles = diarize_batch(
                        self.audio_cache, batch, batch_words, token, profiles, context_end,
                        progress, lambda: self._speaker_checkpoint(nonblocking=True))
                except SpeakerPaused:
                    continue  # Release the CPU slot and retry only this unfinished batch after resume.
                with self.condition:
                    if revision != self.speaker_revision:
                        continue
                    self.speaker_labels = list(self.speaker_labels or [])[:begin] + labels
                    self.speaker_parts = self.speaker_parts[:begin] + parts
                    self.speaker_profiles = profiles
                    self.speaker_done = finish
                    self.content_revision += 1
                    self.save()
        except Stopped:
            with self.condition:
                self.speaker_status = "interrupted"
                self.speaker_phase = "說話者辨識已停止；已完成區段已保存"
                self.speaker_thread = None
                self.save()
        except Exception as error:
            with self.condition:
                self.speaker_status = "failed"
                self.speaker_error = str(error)
                self.speaker_phase = "說話者辨識失敗；可從已保存進度重試"
                self.speaker_thread = None
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
                newly_paused = self.status != "paused"
                self.status = "paused"
                if newly_paused and self.speaker_timing == "pause":
                    self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())
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

    def snapshot(self, include_rows=True):
        with self.condition:
            return dict(name=self.name, status=self.status, phase=self.phase,
                        rows=list(self.rows) if include_rows else [], row_count=len(self.rows),
                        content_revision=self.content_revision, processed=self.processed,
                        preparing_model=self.preparing_model,
                        parallel_enabled=self.parallel_enabled,
                        chunk_parallel_enabled=self.chunk_parallel_enabled,
                        backend=dict(self.backend),
                        word_timestamps=self.word_timestamps,
                        chunk_word_timestamps=self.chunk_word_timestamps,
                        speaker_labels=list(self.speaker_labels or []) if include_rows else [],
                        speaker_done=self.speaker_done,
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

    def export_payload(self):
        with self.condition:
            if self._export_cache is not None and self._export_cache[0] == self.content_revision:
                return self._export_cache[1]
            rows = []
            for i, (a, b, text) in enumerate(self.rows):
                if i < len(self.speaker_parts) and self.speaker_parts[i]:
                    rows.extend((x, y, f"{label or '未確認'}：{word}")
                                for x, y, word, label in self.speaker_parts[i])
                elif self.speaker_done:
                    label = self.speaker_labels[i] if i < len(self.speaker_labels) else None
                    rows.append((a, b, f"{label or '未確認'}：{text}"))
                else:
                    rows.append((a, b, text))
            output = export_rows(rows)
            output['preview'] = export_rows(rows[-300:])['timed']
            output['downloads'] = {key: output[key].encode('utf-8-sig') for key in ('plain', 'timed', 'srt')}
            self._export_cache = (self.content_revision, output)
            return output

    def _recognize_chunk(self, audio, core_start, parallel_enabled, word_timestamps, options):
        while True:
            with inference_slot(self.checkpoint):
                with self.condition:
                    self.preparing_model = True
                if self.backend["engine"] == "whisper.cpp":
                    from whisper_cpp_engine import WhisperCppEngine
                    engine_factory = lambda: WhisperCppEngine(self.backend, self.check_stop)
                    engine = engine_factory()
                else:
                    model = load_model(self.size)
                    if self.mode == "自適應快速":
                        engine_factory = lambda: AdaptiveDecodingPipeline(model, self.check_stop)
                    else:
                        engine_factory = lambda: model if self.mode == "原始精細" else BatchedInferencePipeline(model)
                    engine = engine_factory()
                with self.condition:
                    self.preparing_model = False
                # Do not hold the global slot while paused. In-flight inference finishes its batch.
                if self.pause_requested:
                    segments_list = None
                else:
                    self.check_stop()
                    with self.condition:
                        self.transcribing_since = perf_counter()
                        self.phase = f"轉錄 {timestamp(core_start)} 起的片段（含邊界上下文）"
                    gap = central_silence(audio) if parallel_enabled else None
                    opts = dict(language=self.language, vad_filter=True, word_timestamps=word_timestamps,
                                hotwords=self.terms.strip() or None, **options)
                    if gap is not None:
                        segments = transcribe_pair(audio, gap, engine_factory, opts, self.check_stop)
                    else:
                        segments, _ = engine.transcribe(audio, **opts)
                    try:
                        segments_list = []
                        for segment in segments:
                            self.check_stop()
                            segments_list.append(segment)
                    finally:
                        segments.close()
                        with self.condition:
                            self.stop_transcription_clock()
                del engine, engine_factory
                if self.backend["engine"] != "whisper.cpp":
                    del model
            if segments_list is not None:
                return segments_list
            self.checkpoint()

    def run(self):
        outcome = "completed"
        try:
            with self.condition:
                self.preparing_model = True
                self.phase = "建立共用音訊快取（每份錄音只需解碼一次）"
            duration = self.audio_cache.ensure(self.checkpoint)
            with self.condition:
                self.preparing_model = False
            self.checkpoint()
            if self.speaker_timing == "during":
                self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())
            options = {"快速": {"beam_size": 5, "batch_size": 2},
                       "自適應快速": {"beam_size": 1, "batch_size": 2},
                       "極速": {"beam_size": 1, "batch_size": 2},
                       "原始精細": {"beam_size": 5}}[self.mode]
            converter = OpenCC("s2twp")
            for core_start, core_end, offset, audio in self.audio_cache.windows(duration, self.resume_offset):
                self.checkpoint()
                with self.condition:
                    self.chunk_parallel_enabled = self.parallel_enabled
                    parallel_enabled = self.chunk_parallel_enabled
                    self.chunk_word_timestamps = self.word_timestamps
                    word_timestamps = self.chunk_word_timestamps
                    self.phase = f"等待運算資源：{timestamp(core_start)} 起的片段"
                segments_list = self._recognize_chunk(audio, core_start, parallel_enabled,
                                                     word_timestamps, options)
                for raw_segment in segments_list:
                    segment = owned_segment(raw_segment, offset, core_start, core_end)
                    if segment is None:
                        continue
                    text = converter.convert(segment.text.strip())
                    a, b = max(0, segment.start), min(duration, segment.end)
                    with self.condition:
                        if text and b > a:
                            self.rows.append((a, b, text))
                            self.words.append([(max(0, w.start), min(duration, w.end), w.word)
                                               for w in (segment.words or [])])
                            self.content_revision += 1
                        self.processed = max(self.processed, min(core_end, b))
                        self.condition.notify_all()
                    self.save(force=False)
                    self.checkpoint()
                with self.condition:
                    self.processed = self.resume_offset = core_end
                    self.committed_rows = len(self.rows)
                    self.condition.notify_all()
                self.save()
                del audio, segments_list
                self.checkpoint()
        except Stopped:
            outcome = "stopped"
        except Exception as error:
            outcome = "failed"
            with self.condition:
                self.error = str(error)
        finally:
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
                or (getattr(job, "speaker_thread", None) is not None and job.speaker_thread.is_alive())
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
