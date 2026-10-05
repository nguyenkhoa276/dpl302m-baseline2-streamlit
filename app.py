from pathlib import Path
import tempfile
import streamlit as st
import torch
import librosa
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

st.set_page_config(
    page_title="Baseline 2 - VN to EN",
    layout="wide",
)

ROOT = Path.cwd()
CKPT = ROOT / "checkpoints"

PHO_BASE = "vinai/PhoWhisper-small"
MT_BASE = "vinai/vinai-translate-vi2en-v2"

ASR_FINAL = (
    CKPT
    / "baseline2_phowhisper_small_finetuned"
    / "final_model"
)

MT_ADAPTER = (
    CKPT
    / "baseline2_vinai_translate_vi2en_lora"
    / "final_adapter"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)


@st.cache_resource
def load_asr():
    source = (
        str(ASR_FINAL)
        if ASR_FINAL.exists()
        else PHO_BASE
    )

    processor = AutoProcessor.from_pretrained(source)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        source,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    model.eval()

    return processor, model


@st.cache_resource
def load_mt():
    tok = AutoTokenizer.from_pretrained(MT_BASE)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MT_BASE,
        torch_dtype=DTYPE,
    )

    if MT_ADAPTER.exists():
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                str(MT_ADAPTER),
            )

        except Exception as e:
            st.warning(
                "LoRA adapter could not be loaded. "
                f"Base translation model is used. {e}"
            )

    model = model.to(DEVICE)
    model.eval()

    return tok, model


def transcribe(path, processor, model):
    wav, _ = librosa.load(
        path,
        sr=16000,
        mono=True,
    )

    x = processor(
        wav,
        sampling_rate=16000,
        return_tensors="pt",
    )

    feats = x.input_features.to(
        DEVICE,
        dtype=DTYPE,
    )

    with torch.inference_mode():
        ids = model.generate(
            feats,
            max_new_tokens=225,
        )

    return processor.batch_decode(
        ids,
        skip_special_tokens=True,
    )[0].strip()


def translate(text, tok, model):
    x = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    with torch.inference_mode():
        ids = model.generate(
            **x,
            max_new_tokens=256,
            num_beams=4,
        )

    return tok.batch_decode(
        ids,
        skip_special_tokens=True,
    )[0].strip()


st.title(
    "Baseline 2 — Vietnamese → English Speech Translation"
)

st.caption(
    "PhoWhisper-small + VinAI Translate vi2en-v2"
)

left, right = st.columns(2)


with left:
    st.subheader("Input")

    input_mode = st.radio(
        "Choose input method",
        ["Upload audio", "Record audio"],
        horizontal=True,
    )

    if input_mode == "Upload audio":

        uploaded = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )

        if uploaded:
            st.audio(uploaded)

            run = st.button(
                "Transcribe & Translate",
                type="primary",
                use_container_width=True,
            )
        else:
            run = False

    else:

        recorded_audio = st.audio_input(
            "Record a short Vietnamese conversation"
        )

        if recorded_audio:
            st.audio(recorded_audio)

            run = st.button(
                "Transcribe & Translate",
                type="primary",
                use_container_width=True,
            )
        else:
            run = False


with right:
    st.subheader("Output")

    if run:

        processor, asr_model = load_asr()
        tok, mt_model = load_mt()

        if input_mode == "Upload audio":

            suffix = Path(uploaded.name).suffix or ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:
                tmp.write(uploaded.getbuffer())
                path = tmp.name

        else:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
            ) as tmp:
                tmp.write(recorded_audio.getbuffer())
                path = tmp.name

        with st.spinner("Running PhoWhisper..."):
            transcript = transcribe(
                path,
                processor,
                asr_model,
            )

        st.markdown("**Vietnamese transcript**")
        st.success(transcript)

        with st.spinner("Translating..."):
            result = translate(
                transcript,
                tok,
                mt_model,
            )

        st.markdown("**English translation**")
        st.success(result)

        st.caption(
            f"Device: {DEVICE.upper()} | "
            f"ASR: {'fine-tuned' if ASR_FINAL.exists() else 'pretrained'} | "
            f"MT LoRA: {'loaded' if MT_ADAPTER.exists() else 'base model'}"
        )
