import gradio as gr
import torch
import numpy as np
from soprano import SopranoTTS
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

model = None

def load_model():
    global model
    if model is None:
        model = SopranoTTS(
            backend="transformers",
            device=DEVICE,
            cache_size_mb=100,
            decoder_batch_size=1,
        )
    return model


SAMPLE_RATE = 32000


def tts_generate(text, temperature, top_p, repetition_penalty):
    if text is None or (isinstance(text, str) and not text.strip()):
        return None
    if not isinstance(text, str):
        raise gr.Error("Input text must be a string.")
    for name, value, minimum, maximum in (
        ("Temperature", temperature, 0.0, 1.0),
        ("Top-p", top_p, 0.01, 1.0),
        ("Repetition penalty", repetition_penalty, 1.0, 2.0),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not minimum <= value <= maximum):
            raise gr.Error(f"{name} must be between {minimum} and {maximum}.")

    model = load_model()

    out = model.infer(
        text.strip(),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    audio_np = out.detach().float().cpu().numpy()
    if audio_np.size == 0:
        return None
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    return SAMPLE_RATE, audio_int16


with gr.Blocks(title="Soprano TTS", delete_cache=(3600, 3600)) as demo:

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "# Soprano TTS\n\n"
                "Soprano is an ultra-lightweight, open-source text-to-speech (TTS) model designed for "
                "real-time, high-fidelity speech synthesis at unprecedented speed. Soprano can achieve "
                "**<15 ms streaming latency** and up to **2000x real-time generation**, all while being "
                "easy to deploy at **<1 GB VRAM usage**.\n\n"
                "- GitHub: https://github.com/ekwek1/soprano\n"
                "- Model: https://huggingface.co/ekwek/Soprano-80M"
            )

            text_in = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to synthesize...",
                value="Soprano is an extremely lightweight text to speech model designed to produce highly realistic speech at unprecedented speed.",
                lines=4,
            )

            with gr.Accordion("Advanced options", open=False):
                temperature = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.05, label="Temperature"
                )
                top_p = gr.Slider(
                    0.01, 1.0, value=0.95, step=0.01, label="Top-p"
                )
                repetition_penalty = gr.Slider(
                    1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty"
                )

            gen_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_out = gr.Audio(
                label="Output Audio",
                autoplay=True,
                streaming=False,
                format="wav",
                show_download_button=True,
            )
            gr.Markdown(
                "**Usage tips:**\n\n"
                "- Soprano works best when each sentence is between 2 and 15 seconds long.\n"
                "- Although Soprano recognizes numbers and some special characters, it occasionally "
                "mispronounces them. Best results can be achieved by converting these into their "
                "phonetic form. (1+1 -> one plus one, etc)\n"
                "- If Soprano produces unsatisfactory results, you can easily regenerate it for a new, "
                "potentially better generation. You may also change the sampling settings for more varied results.\n"
                "- Avoid improper grammar such as not using contractions, multiple spaces, etc."
            )

    gen_btn.click(
        fn=tts_generate,
        inputs=[text_in, temperature, top_p, repetition_penalty],
        outputs=[audio_out],
        api_name="generate",
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.queue(api_open=False)
    demo.launch(server_name="127.0.0.1")
