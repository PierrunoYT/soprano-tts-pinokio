# Soprano TTS - Pinokio Setup

A Windows-optimized setup for [Soprano TTS](https://github.com/ekwek1/soprano), an ultra-lightweight, open-source text-to-speech model designed for real-time, high-fidelity speech synthesis.

## Overview

**Soprano** is an incredibly fast TTS model with:
- **Only 80M parameters** (under 1 GB VRAM usage)
- **~2000× real-time factor** - generates 10 hours of audio in under 20 seconds
- **<15 ms latency** for streaming synthesis
- **32 kHz high-fidelity audio** output

## Requirements

- **Windows** (Linux also supported)
- **CUDA-enabled GPU** (CPU support coming soon)
- **Python 3.8+**
- **CUDA 12.6 drivers** installed

## Quick Start (Pinokio)

Simply install via Pinokio - it will automatically:
- Set up the Python environment
- Install PyTorch with CUDA support (via torch.js)
- Install Soprano TTS and all dependencies

### Manual Installation

```bash
# Install dependencies
pip install soprano-tts

# Install PyTorch with CUDA support
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

### Verify Installation

```bash
python check_cuda.py
```

This checks:
- PyTorch installation
- CUDA availability
- Soprano TTS installation

### Run Examples

```bash
python example.py
```

## Usage Examples

### Basic Inference

```python
from soprano import SopranoTTS

# Initialize model
model = SopranoTTS(
    backend='auto',
    device='cuda',
    cache_size_mb=10,
    decoder_batch_size=1
)

# Generate audio
audio = model.infer("Soprano is an extremely lightweight text to speech model.")
```

### Save to File

```python
model.infer("Your text here.", "output.wav")
```

### Custom Sampling Parameters

```python
audio = model.infer(
    "Your text here.",
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2,
)
```

### Streaming Inference (Ultra-Low Latency)

```python
import torch

stream = model.infer_stream("Your text here.", chunk_size=1)

chunks = []
for chunk in stream:
    chunks.append(chunk)  # First chunk arrives in <15 ms!

audio = torch.cat(chunks)
```

### Batched Inference

```python
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]

batch_output = model.infer_batch(texts)

# Save to directory
batch_output = model.infer_batch(texts, "/output_dir")
```

## Performance Tips

### Increase Speed
Adjust these parameters for faster inference (at the cost of higher memory usage):

```python
model = SopranoTTS(
    backend='auto',
    device='cuda',
    cache_size_mb=20,        # Increase from 10
    decoder_batch_size=2     # Increase from 1
)
```

### Best Practices
- Keep sentences between **2-15 seconds** long
- Convert numbers to words: `1+1` → `one plus one`
- Use proper grammar and contractions
- Avoid multiple spaces or special characters
- Regenerate if results are unsatisfactory

## Troubleshooting

### CUDA Not Available
1. Verify CUDA drivers: `nvidia-smi`
2. Check PyTorch CUDA access:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. Reinstall PyTorch with CUDA support:
   ```bash
   pip uninstall -y torch
   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
   ```

### LMDeploy Installation Fails
Use the transformers backend (slower but more compatible):

```python
model = SopranoTTS(backend='transformers', device='cuda')
```

### Unicode/Encoding Errors
The scripts include Windows encoding fixes. If you still encounter issues, ensure your terminal supports UTF-8.

## Project Structure

```
SopranoTTS-Pinokio/
├── pinokio.js          # Pinokio app configuration
├── install.js          # Installation script
├── start.js            # Start script
├── torch.js            # PyTorch installation handler
├── requirements.txt    # Python dependencies
├── check_cuda.py       # System verification script
├── example.py          # Usage examples
└── README.md           # This file
```

## Key Features

### 1. High-Fidelity 32 kHz Audio
Soprano synthesizes speech at 32 kHz, delivering quality that is perceptually indistinguishable from 44.1/48 kHz audio.

### 2. Vocoder-Based Neural Decoder
Uses a Vocos architecture for orders-of-magnitude faster waveform generation compared to diffusion models.

### 3. Seamless Streaming
Leverages the decoder's finite receptive field to losslessly stream audio with ultra-low latency (<15 ms).

### 4. State-of-the-Art Neural Audio Codec
Compresses audio to ~15 tokens/sec at just 0.2 kbps without sacrificing quality.

### 5. Sentence-Level Streaming
Each sentence is generated independently, enabling effectively infinite generation length.

## Limitations

- Trained on 1000 hours of audio (~100× less than other TTS models)
- No voice cloning yet
- No style control yet
- No multilingual support yet
- Requires CUDA GPU (CPU support coming soon)

## Roadmap

- [x] Model and inference code
- [x] Seamless streaming
- [x] Batched inference
- [ ] Command-line interface (CLI)
- [ ] Server / API inference
- [ ] Additional LLM backends
- [ ] CPU support
- [ ] Voice cloning
- [ ] Multilingual support

## Links

- **Official Repository**: https://github.com/ekwek1/soprano
- **HuggingFace Model**: https://huggingface.co/ekwek/Soprano-80M
- **License**: Apache-2.0

## Acknowledgements

Soprano uses and/or is inspired by:
- [Vocos](https://github.com/gemelo-ai/vocos)
- [XTTS](https://github.com/coqui-ai/TTS)
- [LMDeploy](https://github.com/InternLM/lmdeploy)

## License

This project is licensed under the **Apache-2.0** license.

