"""Audio regressions without downloading a model or requiring a GPU."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class AudioTests(unittest.TestCase):
    def setUp(self):
        gradio = MagicMock()
        gradio.Error = ValueError
        spec = importlib.util.spec_from_file_location(
            "tts_app", Path(__file__).resolve().parents[1] / "app" / "app.py"
        )
        self.app = importlib.util.module_from_spec(spec)
        self.splitter = MagicMock(return_value=["hello."])
        with patch.dict(sys.modules, {
            "gradio": gradio, "torch": MagicMock(), "soprano": MagicMock(),
            "soprano.utils": MagicMock(),
            "soprano.utils.text_normalizer": MagicMock(),
            "soprano.utils.text_splitter": MagicMock(
                split_and_recombine_text=self.splitter),
        }):
            spec.loader.exec_module(self.app)
        self.loader = patch.object(self.app, "load_model").start()
        self.addCleanup(patch.stopall)

    def generate(self, text="Hello", temperature=0.3, top_p=0.95, penalty=1.2):
        return self.app.tts_generate(text, temperature, top_p, penalty)

    def output(self, values):
        self.loader.return_value.infer.return_value.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = np.array(values)

    def test_empty_input_does_not_load_model(self):
        for text in (None, "", " \n\t"):
            self.assertIsNone(self.generate(text))
        self.loader.assert_not_called()

    def test_invalid_parameters_do_not_load_model(self):
        for kwargs in ({"text": 3}, {"top_p": 0}, {"top_p": float("nan")},
                       {"temperature": -1}, {"temperature": float("inf")},
                       {"penalty": None}, {"penalty": True}, {"penalty": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.generate(**kwargs)
        self.loader.assert_not_called()

    def test_unspeakable_input_is_rejected_before_loading_model(self):
        self.splitter.return_value = []
        with self.assertRaises(ValueError):
            self.generate("...")
        self.loader.assert_not_called()

    def test_pcm_clips_instead_of_wrapping_and_handles_nonfinite_samples(self):
        self.output([-2, -1, -0.5, 0, 0.5, 1, 2, np.nan, np.inf, -np.inf])
        rate, audio = self.generate(text=" Hello ")
        self.assertEqual(rate, 32000)
        self.assertEqual(audio.dtype, np.int16)
        np.testing.assert_array_equal(audio, [
            -32767, -32767, -16383, 0, 16383, 32767, 32767, 0, 32767, -32767
        ])
        self.loader.return_value.infer.assert_called_once_with(
            "Hello", temperature=0.3, top_p=0.95, repetition_penalty=1.2
        )

    def test_empty_model_output_clears_audio(self):
        self.output([])
        self.assertIsNone(self.generate())


if __name__ == "__main__":
    unittest.main()
