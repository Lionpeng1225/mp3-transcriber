"""Optional local speaker diarization using pyannote Community-1."""


def diarize_transcript(audio_path, rows, token, progress=None):
    try:
        import av
        import numpy as np
        import torch
        from pyannote.audio import Pipeline
    except ImportError as error:
        raise RuntimeError(
            "尚未安裝說話者辨識套件。請在程式資料夾執行："
            ".venv\\Scripts\\python.exe -m pip install -r requirements-speakers.txt"
        ) from error

    if not token or not token.strip():
        raise ValueError("請輸入 Hugging Face 存取權杖")
    if progress:
        progress("載入說話者辨識模型（首次使用會下載）")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token.strip())
    if pipeline is None:
        raise RuntimeError("無法載入模型；請確認已在 Hugging Face 同意模型條款")
    pipeline.to(torch.device("cpu"))
    if progress:
        progress("使用本機音訊解碼器載入錄音")
    # Decode through PyAV (already used by the transcription app) and pass
    # samples directly, avoiding TorchCodec's FFmpeg DLL loading on Windows.
    from audio_chunks import audio_chunks, SAMPLE_RATE
    with av.open(str(audio_path)) as container:
        duration = (container.duration or 0) / av.time_base
    capacity = max(SAMPLE_RATE, int((duration + 2) * SAMPLE_RATE))
    waveform = np.empty(capacity, dtype=np.float32)
    filled = 0
    for _offset, chunk in audio_chunks(audio_path):
        end = filled + len(chunk)
        if end > len(waveform):
            grown = np.empty(max(end, len(waveform) * 2), dtype=np.float32)
            grown[:filled] = waveform[:filled]
            waveform = grown
        waveform[filled:end] = chunk
        filled = end
    samples = torch.from_numpy(waveform[:filled]).unsqueeze(0)
    if progress:
        progress("分析錄音中的說話者（PyAV 解碼）")
    output = pipeline({"waveform": samples, "sample_rate": SAMPLE_RATE})
    turns = output.exclusive_speaker_diarization
    if callable(rows):
        rows = rows()
    labels = []
    for start, end, _text in rows:
        midpoint = (start + end) / 2
        speaker = next((str(label) for turn, label in turns
                        if turn.start <= midpoint < turn.end), None)
        labels.append(speaker)
    return labels
