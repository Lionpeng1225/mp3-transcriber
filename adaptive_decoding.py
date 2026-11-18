"""Use greedy decoding first, then widen the search for uncertain speech."""
from dataclasses import is_dataclass, replace

from faster_whisper import BatchedInferencePipeline
from faster_whisper.transcribe import get_compression_ratio


class AdaptiveDecodingPipeline(BatchedInferencePipeline):
    def __init__(self, model, check_stop):
        super().__init__(model)
        self.check_stop = check_stop

    def generate_segment_batched(self, features, tokenizer, options):
        # The job supplies beam_size=1 for the first pass. The parent forward()
        # aligns the final tokens once, using this original batch's encoder output.
        encoded, outputs = super().generate_segment_batched(features, tokenizer, options)
        self.check_stop()

        def rank(output):
            text = tokenizer.decode(output["tokens"]).strip()
            repetitive = bool(text and get_compression_ratio(text) > options.compression_ratio_threshold)
            return bool(text), not repetitive, output["avg_logprob"]

        retries = []
        for index, output in enumerate(outputs):
            has_text, normal_repetition, score = rank(output)
            if output["no_speech_prob"] <= options.no_speech_threshold and (
                not has_text or not normal_repetition or score < options.log_prob_threshold
            ):
                retries.append(index)

        if retries:
            # faster-whisper releases use either a dataclass or a named tuple.
            wide = (replace(options, beam_size=5) if is_dataclass(options)
                    else options._replace(beam_size=5))
            _, alternatives = super().generate_segment_batched(features[retries], tokenizer, wide)
            self.check_stop()
            for index, alternative in zip(retries, alternatives):
                if rank(alternative) > rank(outputs[index]):
                    outputs[index] = alternative
        return encoded, outputs
