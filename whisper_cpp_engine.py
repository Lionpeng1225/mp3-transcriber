"""Independent whisper.cpp CLI adapter, following Buzz 1.4.5's engine protocol."""
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import wave

import numpy as np


class WhisperCppEngine:
    def __init__(self, config, stop_check):
        self.cli = Path(config["cli"]).expanduser().resolve()
        self.model = Path(config["model"]).expanduser().resolve()
        self.gpu = config.get("gpu", False)
        self.stop_check = stop_check
        if not self.cli.is_file() or not self.model.is_file():
            raise FileNotFoundError("請指定 whisper-cli.exe 與 Whisper.cpp 的 ggml 模型檔")

    def transcribe(self, audio, language=None, hotwords=None, **options):
        return self.segments(audio, language, hotwords), None

    def segments(self, audio, language, hotwords):
        # Keep temporary data on the same drive as the app, rather than the system drive.
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            source = Path(temporary) / "chunk.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
            command = [str(self.cli), "--model", str(self.model), "--language", language or "auto",
                       "--threads", "4", "--max-context", "0", "--suppress-nst",
                       "--entropy-thold", "2.8", "--output-json", "-f", str(source)]
            if not self.gpu:
                command.append("--no-gpu")
            if hotwords:
                command.extend(["--prompt", hotwords])
            vad = self.cli.parent / "ggml-silero-v6.2.0.bin"
            if vad.is_file():
                command.extend(["--vad", "--vad-model", str(vad)])
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                while True:
                    self.stop_check()
                    try:
                        _, error = process.communicate(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if process.returncode:
                    raise RuntimeError(f"Whisper.cpp 執行失敗（{process.returncode}）：" + error.decode('utf-8', errors='replace')[-4000:])
                data = json.loads(Path(str(source) + ".json").read_text(encoding="utf-8"))
                if "transcription" not in data:
                    raise ValueError("Whisper.cpp 未輸出預期的 transcription 結果")
                for row in data["transcription"]:
                    yield SimpleNamespace(start=row["offsets"]["from"] / 1000,
                                          end=row["offsets"]["to"] / 1000, text=row["text"])
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()
