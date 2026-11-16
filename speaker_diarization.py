"""Contextual diarization, persistent speaker prototypes, overlap/word alignment."""
import copy
import numpy as np

_PIPELINE = None


def load_speaker_pipeline(token):
    # Caller holds the shared inference slot: one instance, never used concurrently.
    global _PIPELINE
    if _PIPELINE is None:
        import torch
        from pyannote.audio import Pipeline
        torch.set_num_threads(4)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token.strip())
        if pipeline is None:
            raise RuntimeError("無法載入模型；請確認已同意模型條款")
        pipeline.to(torch.device("cpu"))
        _PIPELINE = pipeline
    return _PIPELINE


def overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def best_label(start, end, turns):
    scores = {}
    for a, b, label in turns:
        scores[label] = scores.get(label, 0) + overlap(start, end, a, b)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return None
    if len(ranked) > 1 and ranked[1][1] >= ranked[0][1] * .8:
        return None
    return ranked[0][0]


def match_speakers(output, profiles, context_start, focus_start, focus_end):
    profiles = copy.deepcopy(profiles)
    names = output.speaker_diarization.labels()
    embeddings = output.speaker_embeddings
    mapping = {}
    if embeddings is None:
        return mapping, profiles
    turns = [(float(t.start) + context_start, float(t.end) + context_start, str(s))
             for t, s in output.exclusive_speaker_diarization]
    candidates, vectors, weights = [], {}, {}
    for i, name in enumerate(names):
        name = str(name)
        vector = np.asarray(embeddings[i], dtype=np.float32)
        norm = np.linalg.norm(vector)
        seconds = sum(overlap(a, b, focus_start, focus_end) for a, b, s in turns if s == name)
        if not np.isfinite(norm) or norm < 1e-8 or seconds <= 0:
            continue
        vector = vector / norm
        vectors[name], weights[name] = vector, seconds
        scores = sorted([(float(np.dot(vector, p['vector'])), key) for key, p in profiles.items()
                         if len(p['vector']) == len(vector)], reverse=True)
        if scores and scores[0][0] >= .75:
            if len(scores) == 1 or scores[0][0] - scores[1][0] >= .08:
                candidates.append((scores[0][0], name, scores[0][1]))
        elif not scores or scores[0][0] < .60:
            if seconds >= 3:
                mapping[name] = 'new'
        # Ambiguous matches and short unknown speech remain unconfirmed.
    used = set()
    for score, name, key in sorted(candidates, reverse=True):
        if key not in used:
            mapping[name] = key
            used.add(key)
    for name, key in list(mapping.items()):
        vector, weight = vectors[name], min(weights[name], 30.0)
        if key == 'new':
            key = f'說話者 G{len(profiles) + 1:03}'
            profiles[key] = {'vector': vector.tolist(), 'weight': weight}
            mapping[name] = key
        elif weight >= 3:
            previous = profiles[key]
            old_weight = min(previous['weight'], 120.0)
            mixed = np.asarray(previous['vector']) * old_weight + vector * weight
            mixed /= max(float(np.linalg.norm(mixed)), 1e-8)
            profiles[key] = {'vector': mixed.tolist(), 'weight': old_weight + weight}
    return mapping, profiles


def diarize_batch(cache, rows, words, token, profiles, context_end, progress, checkpoint):
    import torch
    from opencc import OpenCC
    from audio_chunks import SAMPLE_RATE
    from resource_scheduler import inference_slot
    focus_start, focus_end = rows[0][0], max(r[1] for r in rows)
    # Up to 90s of past speech plus the current ~30s; never look beyond transcribed audio.
    context_end = max(focus_end, min(context_end, focus_end + 15))
    context_start = max(0, min(focus_start, context_end - 120))
    samples = cache.read(context_start, context_end)
    if not len(samples):
        raise ValueError('分析範圍沒有音訊')
    checkpoint()
    with inference_slot(checkpoint):
        pipeline = load_speaker_pipeline(token)
        def hook(step_name, *args, **kwargs):
            checkpoint()
            progress(f'說話者分析：{step_name}')
        output = pipeline({'waveform': torch.from_numpy(samples).unsqueeze(0),
                           'sample_rate': SAMPLE_RATE}, hook=hook)
        checkpoint()
    mapping, profiles = match_speakers(output, profiles, context_start, focus_start, focus_end)
    turns = [(float(t.start) + context_start, float(t.end) + context_start, mapping.get(str(s)))
             for t, s in output.exclusive_speaker_diarization]
    labels, parts = [], []
    converter = OpenCC("s2twp")
    for (start, end, text), row_words in zip(rows, words):
        labels.append(best_label(start, end, turns))
        groups = []
        if row_words:
            for a, b, word in row_words:
                label = best_label(a, b, turns)
                if groups and groups[-1][3] == label:
                    groups[-1][1] = b
                    groups[-1][2] += word
                else:
                    groups.append([a, b, word, label])
        for group in groups:
            group[2] = converter.convert(group[2])
        parts.append(groups)
    return labels, parts, profiles
