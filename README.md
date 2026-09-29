# Soprano TTS for Pinokio

A local Gradio interface for [Soprano TTS](https://github.com/ekwek1/soprano).
Enter English text, adjust sampling settings, and generate 32 kHz WAV audio.
This interface uses the Transformers backend and returns complete clips;
upstream streaming benchmarks do not describe this UI's latency.

## Install and run

1. Install this repository in Pinokio and select **Install**.
2. Once installation checks finish, select **Start**, then **Open Web UI**.
3. Enter text and select **Generate**. Use the audio player's download control
   to save the WAV file.

The first generation downloads model weights from Hugging Face. Later requests
reuse the loaded model. Empty input clears the player without loading the model.
The server listens on `127.0.0.1`; Gradio chooses an available port. Use the URL
shown in the terminal when connecting a client.

The installer creates a Python 3.11 environment under `app/env`, installs the
dependencies, selects a PyTorch build, and checks dependency consistency and app
imports. Only then is the environment marked ready. Existing installations made
before this readiness check need to run **Install** once again.

### Hardware

- Windows and Linux NVIDIA systems use the CUDA 12.8 PyTorch build and need a
  compatible NVIDIA driver. The app falls back to CPU if CUDA is unavailable.
- Windows AMD systems and systems without a supported GPU use CPU inference.
  DirectML is not used by this app.
- Linux AMD installation selects ROCm 6.3. Hardware and driver compatibility are
  required; upstream still lists ROCm support as unfinished. This route has not
  been validated here.
- The macOS installer targets Apple Silicon, using CPU inference in this UI.
  Intel macOS is not supported by the pinned PyTorch 2.7 wheels.

See the [upstream installation guide](https://github.com/ekwek1/soprano#installation)
for model and hardware details. CPU inference is supported; a CUDA GPU is optional.

### Maintenance

- **Update** fast-forwards the launcher repository and reruns installation.
  If Git reports divergent history, resolve it manually before retrying.
- **Install** repairs an incomplete environment without deleting it.
- **Reset** removes only `app/env`. Reinstall afterward. It does not remove the
  shared Hugging Face model cache or files you have downloaded.
- **Save Disk Space** invokes Pinokio's virtual-environment deduplication.

Generated audio is stored in Gradio's cache, with hourly cleanup of files older
than one hour. Download clips you want to keep.

### Manual installation

From the repository root, using Python 3.11:

```sh
python -m venv app/env
# Windows PowerShell:
app/env/Scripts/Activate.ps1
# Linux/macOS instead: source app/env/bin/activate
python -m pip install uv
uv pip install -r app/requirements.txt

# CPU build (for NVIDIA, replace /cpu with /cu128):
uv pip install --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip check
cd app
python app.py
```

Manual installation does not create Pinokio's readiness marker. Use **Install**
if you want to manage that environment through Pinokio.

## API

The running UI exposes a queued Gradio endpoint named `/generate`, with inputs in
this order: `text` (at most 5000 characters), `temperature` (0–1), `top_p` (0.01–1), and
`repetition_penalty` (1–2). Requests share a single inference queue. The response
contains one audio file; blank text returns no audio. There is no separate
download API or session-state argument.

Replace `http://127.0.0.1:7860` in these examples with the running server URL.

### Python

Install `gradio_client`, then:

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
audio_path = client.predict("Hello from Soprano.", 0.3, 0.95, 1.2,
                            api_name="/generate")
print(audio_path)  # Local WAV downloaded by the client
```

### JavaScript

Install `@gradio/client`, then:

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");
const result = await client.predict("/generate", [
  "Hello from Soprano.", 0.3, 0.95, 1.2
]);
console.log(result.data[0]); // Audio file metadata, including its URL
```

### curl

Submit a request (POSIX shell syntax):

```sh
curl -X POST http://127.0.0.1:7860/gradio_api/call/generate \
  -H 'Content-Type: application/json' \
  -d '{"data":["Hello from Soprano.",0.3,0.95,1.2]}'
```

Copy the returned `event_id`, then read the result stream:

```sh
curl -N http://127.0.0.1:7860/gradio_api/call/generate/EVENT_ID
```

The `complete` event contains audio metadata. Download its `url` using
`curl -L 'AUDIO_URL' --output speech.wav`.

## Troubleshooting

- **Install still appears:** inspect the installation terminal for dependency or
  import errors, then retry **Install**. An environment directory alone is not
  proof of successful installation.
- **Unexpected CPU inference:** check `torch.cuda.is_available()` inside
  `app/env`. Verify the NVIDIA driver and rerun **Install** to restore the
  platform-specific PyTorch build.
- **First generation fails:** check the terminal and connectivity to Hugging
  Face. Model weights are downloaded lazily, not during installation.
- **Poor pronunciation:** spell out numbers or symbols and use clear English
  sentences. Sampling settings can change results. This UI has no voice cloning
  or language-selection controls.

## Development checks

```sh
node --test tests/launcher.test.js
python -m unittest discover -s tests -v
```

Audio unit tests require NumPy and mock the model. UI integration tests require
the app dependencies and use deterministic audio without downloading weights.
Launcher tests cover failed-install recovery, maintenance menus, URL capture,
and platform routing. GPU inference and Pinokio execution require separate
runtime validation on the target hardware.

Soprano is distributed under the
[upstream Apache-2.0 license](https://github.com/ekwek1/soprano/blob/main/LICENSE).

