"""Incremental speaker analysis with cooperative pause checkpoints."""


def load_speaker_pipeline(token):
    import torch
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token.strip())
    if pipeline is None:
        raise RuntimeError("無法載入模型；請確認已在 Hugging Face 同意模型條款")
    pipeline.to(torch.device("cpu"))
    return pipeline


def diarize_transcript(audio_path, rows, token, progress=None, *, pipeline=None,
                       checkpoint=lambda: None):
    import numpy as np
    import torch
    from audio_chunks import audio_chunks, SAMPLE_RATE
    if not rows:
        return []
    checkpoint()
    if pipeline is None:
        pipeline = load_speaker_pipeline(token)
    first = round(min(row[0] for row in rows) * SAMPLE_RATE)
    last = round(max(row[1] for row in rows) * SAMPLE_RATE)
    pieces = []
    chunks = audio_chunks(audio_path)
    try:
        for offset, chunk in chunks:
            checkpoint()
            base = round(offset * SAMPLE_RATE)
            lo, hi = max(first - base, 0), min(last - base, len(chunk))
            if hi > lo:
                pieces.append(chunk[lo:hi].copy())
            if base + len(chunk) >= last:
                break
    finally:
        chunks.close()
    if not pieces:
        raise ValueError("已轉錄區段沒有可讀取的音訊")
    samples = torch.from_numpy(np.concatenate(pieces)).unsqueeze(0)
    checkpoint()
    def hook(step_name, *args, **kwargs):
        checkpoint()
        if progress:
            progress(f"說話者分析：{step_name}")
    output = pipeline({"waveform": samples, "sample_rate": SAMPLE_RATE}, hook=hook)
    checkpoint()
    turns = list(output.exclusive_speaker_diarization)
    return [next((str(label) for turn, label in turns
                  if turn.start <= (a + b) / 2 - first / SAMPLE_RATE < turn.end), None)
            for a, b, _ in rows]
