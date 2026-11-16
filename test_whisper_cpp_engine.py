import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from whisper_cpp_engine import WhisperCppEngine


class CppTests(unittest.TestCase):
    def test_protocol_and_timestamps(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            cli = Path(directory) / 'whisper-cli.exe'
            model = Path(directory) / 'ggml.bin'
            cli.touch()
            model.touch()
            engine = WhisperCppEngine(dict(cli=str(cli), model=str(model)), lambda: None)
            commands = []

            class Process:
                returncode = 0

                def __init__(self, command, **kwargs):
                    commands.append(command)
                    wav = Path(command[command.index('-f') + 1])
                    Path(str(wav) + '.json').write_text(json.dumps({'transcription': [
                        {'offsets': {'from': 250, 'to': 1250}, 'text': '中文測試'}
                    ]}, ensure_ascii=False), encoding='utf-8')

                def communicate(self, **kwargs):
                    return b'', b''

                def poll(self):
                    return 0

            with patch('whisper_cpp_engine.subprocess.Popen', Process):
                segments, _ = engine.transcribe(np.zeros(32000), language='zh', hotwords='多巴胺')
                rows = list(segments)
            self.assertEqual(rows[0].start, 0.25)
            self.assertEqual(rows[0].end, 1.25)
            self.assertEqual(rows[0].text, '中文測試')
            self.assertIn('--no-gpu', commands[0])
            self.assertIn('多巴胺', commands[0])

    def test_missing_engine_rejected(self):
        with self.assertRaises(FileNotFoundError):
            WhisperCppEngine(dict(cli='missing.exe', model='missing.bin'), lambda: None)


if __name__ == '__main__':
    unittest.main()
