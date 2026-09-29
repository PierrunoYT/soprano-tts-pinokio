"""Exercise real Gradio serialization and its public API without model downloads."""
import asyncio
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import wave

DEPS_AVAILABLE = all(importlib.util.find_spec(name) is not None
                     for name in ("gradio", "torch", "soprano"))


@unittest.skipUnless(DEPS_AVAILABLE, "Install app requirements to run UI integration tests")
class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}):
            spec = importlib.util.spec_from_file_location(
                "ui_app", Path(__file__).resolve().parents[1] / "app" / "app.py"
            )
            cls.app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.app)
        cls.app.demo.queue(api_open=False)

    def test_public_api_has_no_hidden_state_or_download_dependency(self):
        endpoint = self.app.demo.get_api_info()["named_endpoints"]["/generate"]
        self.assertEqual(len(endpoint["parameters"]), 4)
        self.assertEqual(len(endpoint["returns"]), 1)
        self.assertTrue(self.app.audio_out.show_download_button)
        self.assertFalse(self.app.demo.api_open)

    def test_generation_produces_downloadable_wav_and_blank_input_clears_it(self):
        import torch
        fn = next(fn for fn in self.app.demo.fns.values() if fn.api_name == "generate")
        self.assertEqual(fn.concurrency_limit, 1)
        with patch.object(self.app, "load_model") as loader:
            loader.return_value.infer.return_value = torch.tensor([-2.0, 0.0, 2.0])
            result = asyncio.run(self.app.demo.process_api(
                fn, ["Hello", 0.3, 0.95, 1.2]
            ))
            audio = result["data"][0]
            with wave.open(audio["path"], "rb") as wav:
                self.assertEqual(wav.getframerate(), 32000)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getnframes(), 3)
            loader.reset_mock()
            empty = asyncio.run(self.app.demo.process_api(
                fn, [" ", 0.3, 0.95, 1.2]
            ))
            self.assertEqual(empty["data"], [None])
            for text in ("...", "…", "?", "😀"):
                with self.subTest(text=text), self.assertRaises(self.app.gr.Error):
                    asyncio.run(self.app.demo.process_api(fn, [text, 0.3, 0.95, 1.2]))
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
